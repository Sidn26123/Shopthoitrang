package ptithcm.dao;

import java.util.List;

import javax.transaction.Transactional;

import org.hibernate.Query;
import org.hibernate.Session;
import org.hibernate.SessionFactory;
import org.hibernate.Transaction;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Repository;

import ptithcm.entity.YeuThichEntity;

@Transactional
@Repository
public class YeuThichDAOImpl implements YeuThichDAO{
	
	@Autowired
	SessionFactory factory;
	@Override
	public List<YeuThichEntity> layDSYeuThichCuaUser(int maNd) {
		// TODO Auto-generated method stub
		Session session = factory.getCurrentSession();
		String hql = "FROM YeuThichEntity where nguoiDung.maNd=:maNd";
		Query query = session.createQuery(hql);
		query.setParameter("maNd", maNd);
		List<YeuThichEntity> yeuThichList = query.list();
		
		return yeuThichList;
	}

	@Override
	public YeuThichEntity layYeuThichTheoMaNDVaSanPham(int maNd, String maSp) {
		// TODO Auto-generated method stub
		Session session = factory.getCurrentSession();
		String hql = "FROM YeuThichEntity WHERE nguoiDung.maNd = :maNd AND sanPham.maSP=:maSp ";
		Query query = session.createQuery(hql);
		query.setParameter("maNd", maNd);
		query.setParameter("maSp", maSp);
		YeuThichEntity yt = (YeuThichEntity) query.uniqueResult();
		return yt;
	}

	@Override
	public void addYeuThich(YeuThichEntity yeuThich) {
		// TODO Auto-generated method stub
		Session session = factory.openSession();
		Transaction t = session.beginTransaction();
		try {
			session.save(yeuThich);
			t.commit();
		} catch (Exception ex) {
			// TODO: handle exception
			t.rollback();
			System.out.println("loi add yeu thich");
		} finally {
			session.close();
		}
	}

	@Override
	public void deleteYeuThich(int maYT) {
		// TODO Auto-generated method stub
		Session session = factory.getCurrentSession();
		String hql = "DELETE FROM YeuThichEntity WHERE id =:maYT";
		Query query = session.createQuery(hql);
		query.setParameter("maYT", maYT);
		int affectedRows = query.executeUpdate(); 
		
	}

}
