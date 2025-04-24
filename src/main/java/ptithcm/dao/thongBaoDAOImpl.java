package ptithcm.dao;

import java.util.Date;
import java.util.List;

import javax.transaction.Transactional;

import org.hibernate.Query;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.Transaction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import ptithcm.entity.SanPhamEntity;
import ptithcm.entity.ThongBaoEntity;
import ptithcm.entity.NguoiDungEntity;
import ptithcm.dao.nguoiDungDao;
@Transactional
@Repository
public class thongBaoDAOImpl implements thongBaoDAO{
	
	@Autowired
	private SessionFactory sessionFactory;
	

	@Autowired
	private nguoiDungDao ndDAO;
	
	@Override
	public List<ThongBaoEntity> layThongBaoCuaUser(int mand){
		Session session = sessionFactory.getCurrentSession();
		String hql = "FROM ThongBaoEntity tb WHERE tb.nguoiDung.maNd = :mand";
		
		Query query = session.createQuery(hql);

		query.setParameter("mand", mand);

	    List<ThongBaoEntity> tbND = query.list();
		System.out.println("ma nd count: " + tbND.size());
		return tbND;
	}
	
	@Override
	public void addThongBao(ThongBaoEntity thongBao) {
		Session session = sessionFactory.getCurrentSession();
		Transaction t = session.beginTransaction();
		try {
			
			
			session.update(thongBao);
			t.commit();

		} catch (Exception ex) {
			t.rollback();
			System.out.print("Có lỗi xảy ra khi thêm thông báo vào CSDL");

		} finally {
			session.close();
		}
	}
	
	@Override
	public void deleteThongBao(ThongBaoEntity thongBao) {
		sessionFactory.getCurrentSession().delete(thongBao);
	}
	
	@Override
	public void updateThongBao(ThongBaoEntity thongBao) {
		Session session=sessionFactory.openSession();
		Transaction t = session.beginTransaction();
		try {
			session.update(thongBao);
			t.commit();

		} catch (Exception ex) {
			t.rollback();
			System.out.print("loi");

		} finally {
			session.close();
		}
	}
	
	@Override
	public ThongBaoEntity LayThongBaoTheoMaTB(int maTB) {
		ThongBaoEntity thongBao = (ThongBaoEntity) sessionFactory.getCurrentSession().get(ThongBaoEntity.class, maTB);
		return thongBao;
	}
	
	@Override
	public void markAllNotificationRead(int mand) {
		Session session = sessionFactory.getCurrentSession();
	    // Tạo câu lệnh HQL UPDATE
	    String hql = "UPDATE ThongBaoEntity tb SET tb.isRead = true WHERE tb.nguoiDung.maNd = :mand";
	    
	    // Tạo đối tượng Query từ câu lệnh HQL
	    Query query = session.createQuery(hql);
	    
	    // Thiết lập tham số
	    query.setParameter("mand", mand);
	    
	    // Thực thi câu lệnh UPDATE
	    int rowCount = query.executeUpdate();
	    
	    System.out.println("Số dòng đã được cập nhật: " + rowCount);
	}
}