package ptithcm.dao;

import java.util.List;

import javax.transaction.Transactional;

import org.hibernate.Query;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.Transaction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import ptithcm.entity.GioHangEntity;

@Transactional
@Repository
public class gioHangDaoImpl implements gioHangDao{

	@Autowired
	SessionFactory factory;
	
	
	@Override
	public List<GioHangEntity> layGioHangCuaUser(int maNd) {
		Session session = factory.getCurrentSession();
        String hql = "FROM GioHangEntity where nguoiDung.maNd=:maNd order by sanPham.tenSanPham";
       
        Query query = session.createQuery(hql);
        query.setParameter("maNd",maNd);
        List<GioHangEntity> gioHangList = query.list();
     
		return gioHangList;
	}

	@Override
	public GioHangEntity layGioHangTheoMaNdVaSanPham(int maNd, String maSp) {
		Session session = factory.getCurrentSession();
		String hql = "FROM GioHangEntity WHERE nguoiDung.maNd = :maNd AND sanPham.maSP=:maSp ";
		Query query = session.createQuery(hql);
		query.setParameter("maNd", maNd);
		query.setParameter("maSp", maSp);
		GioHangEntity gh = (GioHangEntity) query.uniqueResult();
		return gh;
	}
	
	@Override
	public List<GioHangEntity> layAllGioHang(){
		Session session = factory.getCurrentSession();
		String hql = "FROM GioHangEntity";
	    Query query = session.createQuery(hql);
	    List<GioHangEntity> listGioHang = query.list();
	    return listGioHang;
	}

	@Override
	public void addGioHang(GioHangEntity gioHang) {
		Session session = factory.openSession();
		Transaction t = session.beginTransaction();
		try {
			session.save(gioHang);
			t.commit();
		


		} catch (Exception ex) {
			t.rollback();
			System.out.print("loi");

		} finally {
			session.close();

		}

		
	}

	@Override
	public void updateGioHang(GioHangEntity gioHang) {
		Session session = factory.openSession();
		Transaction t = session.beginTransaction();
		try {
			session.update(gioHang);
			t.commit();
			

		} catch (Exception ex) {
			t.rollback();
			System.out.print("loi");

		} finally {
			session.close();

		}
		
	}

	@Override
	public void updateSoLuong(int soLuong,int maGh) {
		Session session = factory.getCurrentSession();
		String hql = "UPDATE GioHangEntity SET soLuong =:soLuong WHERE maGh =:maGh";
		Query query = session.createQuery(hql);
		query.setParameter("soLuong", soLuong);
		query.setParameter("maGh", maGh);
		int affectedRows = query.executeUpdate();
		
	}
	
	@Override
	public void updateSize(String size, int maGh) {
	    Session session = factory.getCurrentSession();
	    String hql = "UPDATE GioHangEntity SET size = :size WHERE maGh = :maGh"; // tên tham số đã được sửa
	    Query query = session.createQuery(hql);
	    query.setParameter("size", size); // tên tham số đã được sửa
	    query.setParameter("maGh", maGh);
	    int affectedRows = query.executeUpdate();
	}


	@Override
	public void deleteGioHang(int maGh) {
		Session session = factory.getCurrentSession();
		String hql = "DELETE FROM GioHangEntity WHERE maGh =:maGh";
		Query query = session.createQuery(hql);
		query.setParameter("maGh", maGh);
		int affectedRows = query.executeUpdate();
		
	}}